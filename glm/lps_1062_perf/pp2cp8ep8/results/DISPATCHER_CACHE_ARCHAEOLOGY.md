# DISPATCHER CACHE ARCHAEOLOGY — where the Aug-9 host-sync caches live (gauss, 2026-08-13 ~00:4x CDT)

Lane assignment from borel: reconcile "BT_DSA_CP_LAYOUT_CACHE warnings in
tonight's boot log" vs "flags exist nowhere in the trainer repo", find the
Aug-9 code, and stage a ported patch if recoverable.

**TL;DR: The code was never committed to any git ref anywhere. It survives,
complete and byte-verified, as (a) uncommitted working-tree changes in the
vendored Megatron-LM clone on the shared CPFS mount — which the trainer venv
imports directly via an editable install, on BOTH boxes — and (b) the original
patch files in the Aug-9 run artefacts on the Mac. The B+F pair (the two the
Aug-9 fleet marked SHIP, +11–13% @131k) is fully recoverable; the ported patch
is staged at `results/BF_hostcache_port_mcore_57efae08b.patch` (apply-checked
against the exact mcore pin the 73c24b00 lineage uses). On box A/B the A/B
needs zero code staging — arm = two env vars before `start_trainer.sh`.**

## 1. The contradiction, resolved

The mechanism chain, each link verified tonight on box w56lorq:

1. `server/uv.lock` pins `megatron-core` as
   `source = { editable = "vendor/megatron-bridge/3rdparty/Megatron-LM" }` —
   an **editable** install, not a wheel.
2. The venv's editable finder
   (`server/.venv/.../__editable___megatron_core_0_19_0_d3932e757_finder.py`)
   maps `megatron.core` →
   `/root/.cache/user_artifacts/trainers_main/server/vendor/megatron-bridge/3rdparty/Megatron-LM/megatron/core`
   — the vendored **working tree**, whatever it currently contains.
3. That working tree is at commit `57efae08b` (the exact pin the 73c24b00
   lineage uses: trainers → megatron-bridge @ 20fcf2ea → Megatron-LM @
   57efae08b; both submodule levels pointer-clean) but carries **uncommitted,
   never-committed modifications**: `git diff` = +1652/−26 over 8 files plus
   untracked `megatron/core/lookahead_checkpoint.py`.
4. `/root/.cache/user_artifacts` is a **CPFS network mount**
   (ap-southeast-7 aliyuncs), shared across boxes. Verified on box B
   (3m9o7kq): same commit, same 8 dirty files, same hooks. One dirty tree
   serves both boxes.
5. Every gate logs its state once per process at WARNING by design (the
   Aug-9 v1-inert-patch lesson). Tonight's clean-init boot log
   (`logs/pp2cp8_cleaninit_saveprobe_hang_20260813.log`) shows all seven
   gates "present but DISABLED" ×16 ranks — the whole stack is inert
   tonight; the 918/1052 numbers are uncontaminated.

Why borel's grep found nothing: Mac-side checkouts of the same submodule
pointer (main repo, all four wt-pp2-* worktrees — all verified tonight) are
clean at `57efae08b`. Same commit, different working-tree content. The patch
stack was applied to the box tree with `git apply` on Aug 9–10 and never
committed; the CPFS mount preserved it across box reclaim.

Additional negative results (the "nowhere in git" half):
- `~/Documents/trainers`: no hits at HEAD, in any local/remote branch tip,
  in reflog (Aug 4–11 scanned), or in any of the 6 stashes.
- `~/Documents/Megatron-LM` (the fork clone): no hits in the working tree or
  recent history of any remote ref; the Aug-10 fleet's estate notes
  (`ESTATE_NOTES_fermi.md`) independently recorded the gates absent at all
  branch tips then.
- The flags exist in exactly two kinds of places: the CPFS box tree, and the
  Mac artefact patch files.

## 2. Where the code lives — full inventory of the box dirty tree

Box tree = `57efae08b` + the following, every hunk verified **byte-identical**
to its Mac artefact patch (diff of diffs, tonight):

| File(s) | Provenance (Mac artefact) | Gate | Aug-9/10 verdict |
|---|---|---|---|
| `.../experimental_attention_variant/dsa.py` | `runs/overnight_20260809_dispatcher_hostsync/patches/0002-fix-b-dsa-cp-layout-cache.patch` | `BT_DSA_CP_LAYOUT_CACHE` | **SHIP** (FIX B) |
| `.../models/common/embeddings/rope_utils.py` | same dir, `0003-fix-f-thd-rope-host-cache.patch` | `BT_THD_ROPE_HOST_CACHE` | **SHIP** (FIX F) |
| `.../experimental_attention_variant/dsa_cudnn_kernels.py` | same dir, `0001-fix-a-dsa-bwd-async-nonempty.patch` (v2) | `BT_DSA_BWD_ASYNC_NONEMPTY` | PARK — measured ≈0 (FIX A) |
| `recompute.py` + `.../moe/token_dispatcher.py` | `runs/overnight_20260809_dispatcher_hostsync/fixc/fixc.patch` | `BT_MOE_DISPATCH_REPLAY_CACHE` (+`_VERIFY`) | FIX C, validated recipe exists |
| `.../moe/router.py` | `fixc/fixc_prime_routing_force.patch` | `BT_MOE_ROUTING_REPLAY_FORCE` | C′ |
| `.../tensor_parallel/mappings.py` + `__init__.py` | Aug-10 overlap work (`runs/overnight_20260810_round3/overlap/patches/`, W1) | `BT_MOE_PROBS_A2A_COMM` | SHIPPED-as-v1; **must stay OFF on multi-EP-group jobs** (SCORECARD_MORNING caveat) |
| `.../lookahead_checkpoint.py` (untracked, 588 lines) | Aug-10 overlap work, W3-v3 | `BT_MOE_LOOKAHEAD_RECOMPUTE` | staged/canary-era |

The aggregate state matches the Aug-10 fleet's own record exactly:
`HANDOFF_BOX_fourier.md` documents "diffstat: +1652/−26 over 8 files +
untracked lookahead_checkpoint.py" — identical to tonight's `git diff --stat`
on w56lorq. The box tree is a time capsule of the Aug-10 fleet state.

Snapshot of the full dirty diff secured Mac-side:
`results/mcore_dirty_w56lorq_20260813.diff` (1998 lines, sha256
`e1e46818dfc684566815fdf482574c08bc275451d4471706dc89d0154eee3653`).

## 3. Is the B+F code complete? Yes.

- **Patches**: canonical copies at
  `runs/overnight_20260809_dispatcher_hostsync/patches/0002` + `0003`
  (combined on-box diff: `box-applied-qr4ggv3-v2.patch`).
- **Parity tests**: `runs/overnight_20260809_dispatcher_hostsync/tests/
  test_dsa_cp_layout_cache_parity.py` (191 PASS Mac-CPU) and
  `test_thd_rope_host_cache_parity.py` (167 PASS Mac-CPU), re-verified
  against the on-box state on Aug-9. They exercise the production
  full-recompute checkpoint-replay context — the same context our PP2 config
  runs.
- **Measured win** (Aug-9, golden EP16/CP16, 131k×d4): 634–645 → ~715
  tok/s/GPU steady (+11–13%); 16k×d32 customer shape: 612 → 726 (+18.5%).
  Loss canary ≤2e-3; memory flat; baseline triple-replicated.
- **Acceptance kit**: `check_acceptance.py` trace-counter checker with
  calibrated baseline/post-patch profiles.
- **Mechanisms** (both bitwise-exact by construction, env-gated default OFF,
  gate-off ⇒ byte-identical path):
  - FIX B: caches the packed-CP layout builder results (query positions +
    KV reorder) on the per-microbatch `packed_seq_params` carrier, keyed on
    argument tuple + object identity of the cu_seqlens inputs. Kills
    26,520→~340 nonzeros/step and ~150k kernel launches/step.
  - FIX F: one host copy per unique cu_seqlens tensor (FIFO 64, keyed on
    identity + `_version` so in-place mutation forces a fresh copy). Kills
    587→4 blocking pageable DtoH copies/step (14.8s → 0.16s CPU).

## 4. The port (staged) + box A/B recipe

**Staged patch:** `results/BF_hostcache_port_mcore_57efae08b.patch`
(sha256 `c126dacc8592af3dc56aa9170992460e0edc93cbc1d25e0b9f357a947f76bf83`)
= 0002 + 0003 concatenated. Verified `git apply --check` clean against a
pristine `57efae08b` checkout (the wt-pp2-packing submodule — the exact pin
the 73c24b00 lineage ships). No adaptation was needed: the patches were
written against this pin and the box tree proves they compose with it.

**PR-ready commit series:** `results/pr_series_bf/` — two git format-patch
commits (0001 = FIX B dsa.py, 0002 = FIX F rope_utils.py) on 57efae08b with
full evidence-citing messages and a marked `<TO FILL TONIGHT>` slot for the
PP2/CP8/EP8 A/B result. Verified: `git am` of the series on pristine
57efae08b applies clean and reproduces the hand-built tree exactly. Author
line is Jack's identity, matching the fleet's commits. Not pushed anywhere
— the vendored-fork PR + submodule pointer bumps are a morning decision.

**On box A or B, the A/B is env-only** (hooks already live in the shared
tree; `start_trainer.sh` uses `srun --export=ALL`):

```bash
export BT_DSA_CP_LAYOUT_CACHE=1 BT_THD_ROPE_HOST_CACHE=1
bash /root/.cache/user_artifacts/.devbox_up/start_trainer.sh   # usual config
```

Arm checks (from the Aug-9 activation protocol): boot log must show
`BT_DSA_CP_LAYOUT_CACHE=1: DSA packed-CP layout cache ACTIVE` and
`BT_THD_ROPE_HOST_CACHE=1: THD RoPE cu_seqlens host cache ACTIVE` (WARNING,
once per process). No ACTIVE line = inert = stop.

**Validation ladder** (standing fleet discipline): d2 canary (loss
12.2–12.4 band, grad-norm comparability, peak mem vs 275 GiB) → d4 A/B vs
918 → d16 A/B vs 1052. Aug-9 acceptance bars for the trace, if one is taken:
`aten::nonzero` ≤ ~200 calls / ≤ 0.2s CPU; blocking pageable-DtoH >1ms = 0.
NOTE: `check_acceptance.py`'s calibrated profiles were built for the 78-layer
PP1 topology — per-rank counts on PP2 (38/40 layers per stage) need
recalibration before those rows are meaningful. The ACTIVE lines + loss
canary are the primary gates.

## 5. Interaction with tonight's PP2/CP8/EP8 M=N config — safe by construction

- Both caches live on the `packed_seq_params` carrier, whose lifetime is one
  microbatch (fwd through its recompute replay). M=N changes the schedule
  (one call, M microbatches), not the per-microbatch carrier lifetime.
  Identity-keyed entries cannot go stale across microbatches.
- Full recompute stays on; the replay context is exactly what the parity
  tests exercise.
- The Aug-9 win was measured at EP16/CP16 on the golden topology. Whether
  the same +10%-class win holds at PP2/CP8/EP8 d16 is precisely what the L3
  trace (§3b pre-registered rules) decides: if CPU-blocked critical-path
  share ≥ 15% of step, this A/B is the top post-L1 priority.

## 6. Caveats

- Keep `BT_DSA_BWD_ASYNC_NONEMPTY` (FIX A) OFF — measured ≈0/−2% on Aug-9
  (dispatcher replay eventSync throttles upstream of it; FIX C is its
  prerequisite).
- W1 (`BT_MOE_PROBS_A2A_COMM`) is in the tree but prohibited on
  multi-EP-group topologies pending the fixed build's validation there
  (Aug-10 SCORECARD). Not part of this A/B.
- The other in-tree levers (FIX C, C′, W3) are separate experiments with
  their own validation state; this lane scoped only B+F.
- Morning PR note: the caches are **not** in the 73c24b00 branch content —
  they ride the CPFS working tree. Any fresh clone (new box, new seeker)
  needs the staged patch applied to the vendored mcore before the env vars
  do anything. The durable fix is an upstream PR to the vendored fork;
  the staged patch + parity tests are the raw material.

## 7. DIRTY-TREE HYGIENE — standing finding + cleanup path

**Finding:** every box measurement this night (918 @d4, 1052 @d16, the
parity legs, the save probes) imported a vendored Megatron-LM working tree
that is pointer-clean (`57efae08b`, matching the 73c24b00 lineage's pin via
megatron-bridge @ 20fcf2ea) but **content-dirty: +1652/−26 across 8 files
plus an untracked 588-line `lookahead_checkpoint.py`** — the uncommitted
Aug-9/10 experiment stack, preserved on the CPFS share that both boxes
mount. The outer trainers repo cannot see this (`git submodule status`
reports the pointer, not the content).

**Env-gate inventory present in the tree (all seven logged DISABLED ×16
ranks on tonight's clean-init boot):**

| Gate | File(s) | Origin | Status |
|---|---|---|---|
| `BT_DSA_CP_LAYOUT_CACHE` | dsa.py | FIX B (0002) | SHIP pair — PR series staged |
| `BT_THD_ROPE_HOST_CACHE` | rope_utils.py | FIX F (0003) | SHIP pair — PR series staged |
| `BT_DSA_BWD_ASYNC_NONEMPTY` | dsa_cudnn_kernels.py | FIX A v2 (0001) | parked (measured ≈0) |
| `BT_MOE_DISPATCH_REPLAY_CACHE` (+`_VERIFY`) | recompute.py, token_dispatcher.py | FIX C (fixc.patch) | validated recipe, untimed |
| `BT_MOE_ROUTING_REPLAY_FORCE` | router.py | C′ (fixc_prime) | experimental |
| `BT_MOE_PROBS_A2A_COMM` | mappings.py, tensor_parallel/__init__.py | W1 (Aug-10) | **prohibited on multi-EP-group jobs** pending fixed-build validation |
| `BT_MOE_LOOKAHEAD_RECOMPUTE` | lookahead_checkpoint.py, recompute.py | W3-v3 (Aug-10) | canary-era |

**Why gate-off is believed inert (gating structure, verified against the
full dirty diff):** (1) every gate is a per-call `os.environ.get(..., "0")
== "1"` read with the original code preserved verbatim in the else branch —
each patch's notes assert and the diff confirms "gate off ⇒ byte-identical
path"; (2) all new symbols are additive-only (new classes/functions/exports
— `all_to_all_deferred`, `wait_deferred_a2a`, `_CheckpointChunkPassMarker`,
`lookahead_checkpoint.py`); no existing symbol is redefined; (3) the single
control-flow edit outside an else branch (recompute.py `if` → `elif`
inserting the lookahead branch) sits behind `_lookahead_checkpoint_enabled()`,
default False; (4) empirical: all seven gates self-reported DISABLED on all
16 ranks tonight, and the night's numbers matched the prediction chain built
on clean-tree expectations. Residual non-inert surface: one WARNING line per
gate per process and per-call env reads — no numerics impact.

**Why this must be cleaned up (Jack's un-owned-complexity directive):**
(a) the CPFS tree is mutable shared state — any `git checkout/reset/clean`
inside that submodule destroys the only live copy of the gates (artefact
patches + tonight's snapshot diff are the backup); (b) a fresh box/seeker
silently boots a different effective codebase — arming the env vars there
does nothing, with no error; (c) the night's PR evidence was measured on
bits a reviewer cannot exactly reconstruct from the branch alone (inert
tonight, but the imprecision is the kind that bites later).

**Recommended cleanup path (in order):**
1. **Land the SHIP pair properly**: PR to the vendored Megatron-LM fork
   from the staged series (`results/pr_series_bf/`, two format-patch
   commits on 57efae08b), then bump the megatron-bridge → Megatron-LM
   pointer and the trainers → megatron-bridge pointer on the 73c24b00
   lineage. This is the fix Jack's directive asks for — the work lands,
   not just the flags.
2. After the pointer bump, `git submodule update` yields a clean tree that
   already contains B+F; the venv (editable install) needs no rebuild.
3. **Revert the rest from the CPFS tree** (`git apply -R` of fixc /
   fixc_prime / W1 / W3 hunks + remove lookahead_checkpoint.py, or
   wholesale `git checkout -- .` inside the submodule) — everything is
   preserved Mac-side (artefact patches + `results/
   mcore_dirty_w56lorq_20260813.diff`). Sequence with doppler between
   boots; all gates are OFF everywhere, but the CPFS tree is currently the
   only runnable copy for the §3b A/B, so the revert waits until that A/B
   is done (or the PR lands and the tree is reset to the bumped pin).
4. The remaining levers (C, C′, W1, W3) each get their own land-or-drop
   decision per their validation state — not part of this queue item.

## Evidence index

- PR-ready series: `results/pr_series_bf/0001-*.patch`, `0002-*.patch`
  (format-patch, `git am`-verified against pristine 57efae08b)
- Box diff snapshot: `results/mcore_dirty_w56lorq_20260813.diff`
- Tonight's gate-disabled lines: box `logs/pp2cp8_cleaninit_saveprobe_hang_20260813.log`
  (all 7 gates × 16 ranks, DISABLED)
- uv.lock editable pin: box `trainers_main/server/uv.lock`
  (`source = { editable = "vendor/megatron-bridge/3rdparty/Megatron-LM" }`)
- Editable finder mapping: box `server/.venv/lib/python3.12/site-packages/__editable___megatron_core_0_19_0_d3932e757_finder.py`
- Aug-9 report/notes: `runs/overnight_20260809_dispatcher_hostsync/REPORT.md`,
  `patches/PATCH_NOTES.md`, `ATTRIBUTION.md`
- Aug-10 stack record: `runs/overnight_20260810_round3/overlap/HANDOFF_BOX_fourier.md`,
  `SCORECARD_MORNING.md`, `results/box1_arm_log.md`
