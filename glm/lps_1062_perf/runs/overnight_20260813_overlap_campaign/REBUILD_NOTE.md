# REBUILD NOTE — B/F host-cache stack + C′/A-v3 landing prep (bohr, 2026-08-13 ~19:2x PDT)

Lane: lever 3 / task 4 (fermi assignment). Mac-side only; no GPU actions, nothing pushed.

## 1. What was rebuilt

**Premise correction:** B/F is not branchless — mcore-side it exists as fork PR
**#26** (`jackrao/lps-1062-ship-bf` = `trainers-main` + 2 commits, DRAFT,
MERGEABLE/CLEAN; `trainers-main` == `57efae08b` exactly, so the PR is 2 commits
on today's main, 0 behind). What was actually missing (gauss's cleanup path,
step 1) is the **runnable pointer chain** from the campaign lineage. Built
locally now:

| repo | branch | commit | content |
|---|---|---|---|
| Megatron-LM | `jackrao/lps-1062-ship-bf` (exists, PR #26) | `500ce306a` | `57efae08b` + FIX B + FIX F + FIX A-v2 (parked, default-OFF) + docstring nit |
| Megatron-Bridge | `jackrao/lps-1062-bf-rebuild` (new, local) | `0e356eb2` | `20fcf2ea` + bump `3rdparty/Megatron-LM` → `500ce306a` |
| trainers | `jackrao/lps-1062-bf-rebuild` (new, local) | `e864115a` | `6fa3bfa7` (= origin/jackrao/lps-1062-pp2cp8ep8 tip) + bump `server/vendor/megatron-bridge` → `0e356eb2` |

**Nothing is pushed.** The pointer commits reference mcore/bridge commits that
exist on the fork remote (ship-bf) resp. only locally (bridge bump). Push =
orchestrator/Jack decision (merge-queue posture).

## 2. Verification (all re-done independently, not transcribed from gauss)

- Artefact patches `0001`(v2)`/0002/0003` `git apply --check` **clean** on
  pristine `57efae08b`.
- `57efae08b` + the three patches vs `origin/jackrao/lps-1062-ship-bf`: total
  delta = **one docstring hunk** in `dsa_cudnn_kernels.py` (the `500ce306a`
  review nit, `all_rows_nonempty` vs `nonempty_rows_verified`). B/F content on
  the branch is byte-identical to the Aug-9 validated patches.
- **M=N conflict check: impossible textually, safe semantically.** The campaign
  branch never touched the bridge pin vs `origin/main` (both `20fcf2ea`); its
  22 changed files are all trainer-side (16 under `server/`, 6 in
  `models/`/`.github`) — no mcore-side content. B/F is mcore-side (`dsa.py`,
  `rope_utils.py`). `73c24b00` (M=N merge) is an ancestor of the new trainers
  branch. Semantic safety (corrects gauss §5's mechanism statement): FIX B's
  cache lives on the per-microbatch `packed_seq_params` carrier (M=N changes
  the schedule, not carrier lifetime); FIX F's RoPE host cache is a
  **process-global** FIFO(64) keyed on tensor identity + `_version` (in-place
  mutation forces a fresh copy) — schedule-independent by construction.
- Chain ls-tree verified link by link; trainers diff vs base = the gitlink only.

## 3. Deviations from the assignment letter

1. Used the existing `ship-bf` branch instead of re-creating one from the patch
   files — it is byte-identical to the patches (verified), so a re-cut would
   only fork provenance from PR #26.
2. The pin carries **FIX A v2 (parked, default-OFF)** along with B/F — removing
   it would diverge from PR #26; gate-off is byte-identical-path inert. Keep
   `BT_DSA_BWD_ASYNC_NONEMPTY` **OFF** (Aug-9: ≈0/−2%, prerequisite is FIX C).
3. Branches are local-only (see §1).

## 4. C′ (PR #27) / A-v3 (PR #29) state

- **PR #27 (C/C′)**: DRAFT, MERGEABLE/CLEAN, stacked on `ship-bf` — composes
  with the campaign pin by construction. `ship-cprime` = ship-bf + FIX C
  (`recompute.py`, `token_dispatcher.py`) + C′ (`router.py`) + review fix
  `5d2ed0d06` (stash keyed by `id(router)`). Cosmetic caveat (reviewer-found):
  `ship-cprime` branched from `ship-bf~1` (`06ff6624d`), so it lacks the
  `500ce306a` docstring nit and PR #27's diff carries a spurious one-hunk
  docstring revert; still MERGEABLE, cosmetic only. No prep needed.
- **PR #29 (A-v3)**: DRAFT, MERGEABLE/CLEAN, but stacked on **W1** (`ship-w1`)
  — and W1 is **prohibited on our topology** (PP2/CP8/EP8 has 2 EP groups; W1
  stays OFF on >1-EP-group topologies pending fixed-build validation, campaign
  orders lever 4). Also found: `ship-w1`/`ship-av3` branched off `ship-cprime`
  **before** the `5d2ed0d06` review fix, so those trees carry the stale
  layer_number-keyed C′ (textually mergeable; the fix survives a merge
  one-sided).
- **Prepped (local, Megatron-LM clone):** `jackrao/lps-1062-ship-av3-wo-w1` =
  `40d8b3578` = `ship-cprime` tip + cherry-picked `d58a3214a` (A-v3). Applied
  **clean** (A-v3 touches only `dsa.py`/`dsa_cudnn_kernels.py`/`dsa_kernels.py`,
  disjoint from W1's files); result = corrected C′ + A-v3, zero W1 content.
  This is the landable A-v3 shape for PP2/CP8/EP8 if A-v3 is ever revived.

## 5. Pre-registered A/B plan (B/F at M=N, fixed wheel)

House rules: canary first, cross-boot x2/x2 arm structure (env is
launch-bound), significance = |Δmean| > max(arm spreads) (d4 noise floor ran
3–4% tonight), memory watch vs the 275 GiB cap (canary protocol) with the
campaign's >255 GiB reserved abort line (E1/L0b ramp discipline) kept on armed
boots, `BT_SAVE_STATE_SYNC=1`.

- **Arm:** `export BT_DSA_CP_LAYOUT_CACHE=1 BT_THD_ROPE_HOST_CACHE=1` before
  `start_trainer.sh`. **Hard gate:** boot log must show both `... ACTIVE`
  WARNING lines ×16 ranks; no ACTIVE line = inert = stop (Aug-9 v1 lesson).
- **Correctness gate (before any perf read):** d2 canary on the armed boot —
  loss 12.2–12.4 band, gn comparable to off-arm.
- **Rung 1 (primary): d4, 524k tok/step** vs fixed-wheel reference ~878–886
  tok/s/GPU. *The smaller-d rung is the campaign-orders bet* (lever 3: "at d16
  mostly absorbed — worth more at smaller d"). Aug-9's +11–13% was
  convoy-regime and does NOT transfer; honest pre-registered EV at d4:
  **+0–5%**. (Tension on record: L3_CPU_BLOCKED_DEEP_DIVE §3 measured
  absorption as structurally M-independent in steady state, holding at d4 —
  this rung adjudicates that against the campaign-orders frame.)
- **Rung 2: d16, 2M tok/step** vs number of record **984** (133.2 s).
  Pre-registered bands, adopting gauss's on-record prediction
  (L3_CPU_BLOCKED_DEEP_DIVE §4, written before any such A/B; its "1052"
  reference was the stale wheel — bands recomputed vs 984): **+3–6% (central
  ~+4%)** if the rank-0 builder-window/RoPE exposure (~7.7 s/step ≈ 5.7%) is
  causal; **floor +0.5–1%** if those gaps are phase-locked to pipeline
  handoffs (then only borel's sync-anchored 0.94 s ≈ 0.7% + launch-drag sliver
  is realized — this floor IS the CAMPAIGN_ORDERS "absorbed at d16" frame);
  **>+6%** = superlinear compounding, re-examine d16 priority; **below floor**
  refutes the builder-window mechanism. The A/B adjudicates which scenario
  holds; both in-band outcomes validate the model.
- **Decision rule:** d4 Δ ≥ +3% with canary PASS → recommend landing (PR #26 +
  the pointer bumps = this branch) and extend a probe to the customer 16k-d32
  shape (Aug-9's biggest win, +18.5%). Null at both rungs → land anyway for
  hygiene (gate-off inert; kills the fresh-clone/dirty-tree hazard) but mark
  perf-unproven under M=N.

### On-record mechanism prediction (added 2026-08-13 ~20:4x PDT, per fermi relay of kepler's IDLE_WINDOW_DECOMPOSITION.md — bands above UNCHANGED)

Kepler's d16 idle decomposition (fixed-wheel trace): `aten::nonzero` census =
**55,328 calls / 33.97 s host time (avg 614 µs)** — matches erdos's
"absorbed" line — **but 6.97 s of it is NOT absorbed: it leaks into pure GPU
idle** (4.06 s as ≥1 ms named stalls; call path `CheckpointFunction(Backward)
→ aten::index → aten::nonzero → cudaStreamSynchronize`, GPU bookends cub
reduce/compare → DeviceSelectSweep — the DSA indexer's data-dependent top-k
count readback, firing fwd 1.52 s + bwd-recompute 2.55 s). The "runahead
absorbs everything" framing was only mostly true.

**Class mapping (on record):** the leaked class's signature — boolean-mask
indexing under the checkpoint replay, fwd + recompute split — matches FIX B's
target class (the `dsa_layout.py` packed-CP layout builders), NOT the parked
FIX A class (bwd-only `nonzero(topk_length>0)` compaction drains, ~123 ms
per-call — a different per-call scale and phase).

**UPDATE (IDLE_RESIDUAL_MAP.md landed, kepler 19:5x): mapping CONFIRMED by
site decomposition — 54,720 / 55,328 (98.9%) of the nonzero census are the CP
layout builders (27,360 fwd + 27,360 recompute-replay) = FIX B's target.**
Coverage: B kills 6.20 s of the 6.97 s leak (F +0.86 s; residual 12.14 s is
new territory, not this stack). Kepler's on-record d16 bands for this A/B
(filed before the run, vs 984 / 133.2 s): **central +4–7% (point +5%;
1,018–1,052 tok/s/GPU)** = direct-leak 7.06 s × conversion 0.7–1.0 + 1–2.5 s
launch relief (B removes ~150–200k launches/step); **floor +1.5–2.5%** if
re-absorption eats the direct leak; **ceiling +9–11%** at 1:1 conversion with
compounding; **< +1.5% refutes** the leak-causality model (escalate). Kepler
notes the central sits ~1 pt above gauss's +3–6% band because the leak is
measured directly on this trace; the bands are consistent. d4 rung: no new
prediction — gauss's +0–5% stands (d4 exposure smaller in absolute terms,
launch-drag share larger).

**Prediction (on record):** B/F-ON at d16 should (a) shrink pure GPU idle by
**up to ~7 s/step** (kepler's leak number; B's share 6.20 s), and (b) cut the
nonzero census by the cached fraction (Aug-9 reference: 26,520 → ~340 ≈ 98.7%
cached at 78L/2mb; tonight's expectation ≈ 54,720 → ~700 by the same
per-microbatch structure). Falsifier: B/F-ON with verified ACTIVE arms that
does NOT shrink the nonzero census by the predicted fraction ⇒ the leaked
class is not FIX B's → re-examine before trusting any perf delta. (fermi's
relay said "B/F+C′"; the arm stays B/F-only per this lane's assignment — C′
arming is a separate call.)
- **If a trace is taken:** `check_acceptance.py`'s calibrated profiles are
  PP1/78-layer — per-rank counts need PP2 (38/40) recalibration before those
  rows mean anything (gauss §4). ACTIVE lines + canary are the primary gates.

## 6. Operational notes for the box mechanic (doppler lane)

- Fresh box from `jackrao/lps-1062-bf-rebuild` **(after the push lands — the
  branch and its bridge/mcore pointer commits exist on no remote today)**:
  recursive submodule update yields mcore with B/F present; editable-install
  venv needs no rebuild. Standing WHEEL rule still first (cudnn-frontend bump
  to 1.27.0.dev20260803).
- On the **current shared CPFS tree** the same B/F content is already present
  (dirty tree, gates OFF) — env-only arm is equivalent for these two gates;
  the branch is the durable path. Do NOT `git checkout/reset/clean` inside the
  CPFS mcore submodule until the pointer bump lands (it is the only live copy
  of the other gates; snapshot:
  `pp2cp8ep8/results/mcore_dirty_w56lorq_20260813.diff`).

## Evidence index

- Chain: trainers `e864115a` → bridge `0e356eb2` → mcore `500ce306a`
  (`jackrao/lps-1062-ship-bf`, PR #26).
- Verification scratch: patched-pristine-`57efae08b` vs ship-bf diff = 1
  docstring hunk (this lane, independent of gauss's `pr_series_bf` check).
- A-v3 restack: `jackrao/lps-1062-ship-av3-wo-w1` = `40d8b3578` (local).
- Background: `pp2cp8ep8/results/DISPATCHER_CACHE_ARCHAEOLOGY.md` (gauss),
  `runs/overnight_20260809_dispatcher_hostsync/patches/PATCH_NOTES.md`.
