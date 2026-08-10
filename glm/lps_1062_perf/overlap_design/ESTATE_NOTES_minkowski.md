# ESTATE NOTES — minkowski (design owner, LPS-1062, from 2026-08-09)

Running log of corrections, rulings, and verification artifacts for the
design estate. Inherited from helmholtz (HANDOFF_DESIGN_helmholtz.md);
orchestrator kepler; box runner fourier; second reviewer hilbert.

---

## 2026-08-09 — CORRECTION: inherited box-state claims unverified

- **Falsified:** helmholtz's "all patches verified apply-clean +
  byte-for-byte" (HANDOFF §2 artifact table) held for Mac↔patch only, NOT
  for the box. fourier's guard-(b) found the box tree carrying **W2 v1**
  (box patch md5 `2b06d11a…`, 865 lines) where v3 (`c94f72e6…`, 967 lines)
  was believed deployed; zero hits for v3 marker comments in the box's
  vendored `token_dispatcher.py`.
- **Standing rule (ratified by kepler):** any inherited box-state claim is
  UNVERIFIED until fourier's full patch-estate md5 sweep (box-vs-Mac, incl.
  staged W3 + A-v3) reports. The artifact table's "verified" column is
  Mac-side provenance only.
- **Consequence:** T2 gate held until W2-v3 surgery (reverse-v1 → verify
  pre-W2 → apply-v3 → verify manifest below) passes by inspection.

## 2026-08-09 — W2 v3 post-surgery verification manifest (issued to fourier)

Patch identity: `w2-moe-a2a-pipeline.patch` md5
`c94f72e62e0431eec915ddcf792f4fa0`, 967 lines, 47334 bytes. Purely additive:
**+810 / −0** across 3 files — `token_dispatcher.py` +713, `moe_layer.py`
+59, `experts.py` +38 (all under `megatron/core/transformer/moe/`).
[CORRECTED per fourier's numstat footnote: my first manifest read
+754 (662/56/36) from an awk `^\+[^+]` count, which excludes added lines
whose content begins with `+` (patch `++` lines) — 56 such lines.
`git apply --numstat` is authoritative: 713/59/38, 0 deletions. Marker
tables below are unaffected — they were counted with `grep -c -F` on tree
files and inclusive `^+` on the patch.]

Every marker below is patch-introduced (patch-adds == Mac-tree count ⇒ zero
pre-existing in base), so each is a clean discriminator. Markers keyed † are
v3-ONLY (hilbert round-1/2 + T2-defect fixes) — they distinguish v3 from v1,
not just W2 from base.

**Phase 1 — after reverse-v1 (pre-W2 state):** all markers ZERO hits in all
three files: `_W2ChunkPlan`, `BT_MOE_A2A_PIPELINE`, `forward_expert_group`,
`w2_local_counts_host`, `w2_global_counts_host`,
`_A2A_PIPELINE_DISARM_LOGGED`, `had_buf`.

**Phase 2 — after apply-v3, expected `grep -c -F` per file:**

| Marker | token_dispatcher.py | moe_layer.py | experts.py |
|---|---:|---:|---:|
| `BT_MOE_A2A_PIPELINE` | 20 | 5 | 1 |
| `_W2ChunkPlan` | 3 | 0 | 0 |
| `w2_local_counts_host` † | 9 | 0 | 0 |
| `w2_global_counts_host` † | 10 | 0 | 0 |
| `_A2A_PIPELINE_DISARM_LOGGED` † | 1 | 3 | 0 |
| `had_buf` † | 2 | 0 | 0 |
| `subset of chunks, not a permutation` † | 1 | 0 | 0 |
| `reason = "overlap_moe_expert_parallel_comm"` † | 1 | 0 | 0 |
| `forward_expert_group` | 0 | 1 | 1 |

**Phase 3 — byte-for-byte tier** (Mac ship-tree file md5s; valid iff the
rest of the estate — base `57efae08b` + FIX A/B/F + FIX C + W1-v1 — matches
box, which the estate sweep covers):

- `token_dispatcher.py` = `3857aeddcd37263252baf630b17ac416`
- `moe_layer.py` = `71301302050b90ffb2ad8ea130ec3f80`
- `experts.py` = `aec9b3b8aafcff7ee7c9c314ce58eba1`

Mismatch with exact markers/diffstat ⇒ drift elsewhere in the file; diff
against the Mac copy to localize. Apply cleanliness: dry-run clean, no
`.rej`/`.orig`, diffstat exactly +810/−0 over the 3 files.

**Surgery outcome (fourier, 2026-08-09): PASS.** Markers 13/13 verbatim,
moe_layer + experts md5 exact, patch identity verbatim. Phase-1 reversal
required `patch -R --fuzz=3` (v1 hunk-1 context spanned bohr's
instrumentation slot lines; `git apply -R` inapplicable); pre-W2 state
proven by W1 reverse-check clean + zero W2 markers + exact diffstat.
Provenance preserved at `w2_surgery_provenance/` on the box (pre-v3 .orig
kept). Gate GO per kepler; fourier executing.

**token_dispatcher.py file-md5 tier — box-expected-different, standing:**
the box legitimately carries `fixc_verify_instrumentation.patch` (4
verify-mode logging blocks) which the Mac ship tree deliberately lacks, so
box token_dispatcher.py md5 ≠ Mac `3857aedd…` by exactly those blocks —
fourier localized the drift by diff, zero W2-content delta. For any future
estate sweep: the md5 mismatch is the EXPECTED state; the diff-localization
is the proof, not the md5. HAZARD (from HANDOFF_BOX_bohr §6): the
instrumentation patch's hunk 3 fuzz-MISPLACES over the W1/W2 tree
(`entry.routing_map_dev = …` before `entry = _ReplayEntry()` →
UnboundLocalError); bohr hand-fixed the placement on-box and the on-box
file is the working authority, but the archive patch remains
misplaced-as-authored pending regeneration — **live owner: boltzmann**
(queued behind his C′ adjudication; hilbert is released — corrected per
kepler, 2026-08-09). Any from-patches tree rebuild must re-apply that
hand-fix or it reintroduces the crash. Hazard stands until the regenerated
patch md5s consistently.

## 2026-08-09 — C′ verdict + W3 watch-list amendment (from boltzmann's adjudication via kepler)

C′ PASS on mechanism (HOLD lifted; A-v3 unlocked in ARM-5). Finding with
direct W3 consequence: under C′-on, the 312 DSA-bwd drains absorb ~24 s of
redistributed CPU wait (was ~19 s with the replay throttle present), 97.4 %
GPU-kernel-covered, wall-flat — **wait conservation, not new work**.

W3 canary watch-list amendments (binding at readout):
1. **Baseline discipline:** the W3 canary's deltas are vs the C′-ON profile
   (ARM-5 baseline), never the pre-C′ anchors. The bwd-phase timeline W3's
   lookahead targets now carries its wait in the DSA-bwd drains, not the
   replay eventSyncs — a readout against pre-C′ profile bounds would
   misattribute C′'s eventSync removal to W3.
2. **Where freed time should surface:** compression of the drain-covered
   bwd windows via kicked-recompute kernels overlapping INTO those windows
   (raising kernel occupancy/concurrency during drains), plus the standing
   mechanism bar (bwd comm overlap ≥50 %, SendRecv in bwd windows). The
   drains are 97.4 % GPU-kernel-covered — they are NOT GPU-idle slack, so
   do not book them as reclaimable idle; the win shows as overlap, not as
   drain removal.
3. **Wait accounting by cause, not total:** trace readout must bucket
   host wait by cause (drains vs eventSyncs vs exposed SendRecv) — totals
   are conserved across C′/W3 and prove nothing alone.

## 2026-08-09 — T2 gate HARD FAIL (v3, fixc-included run) — investigation ledger

Gate run on the manifest-verified v3 tree: **HARD FAIL**. In-process arming
VALIDATED (fixc=1: hits 1→3, misses 0→0, both stats asserts PASS). Two
defect threads, both now REAL code suspects (harness thread closed):

- **Defect 1 — backward numerics:** `W2 input grad == monolithic` FAILS on
  balanced ckpt=0/1, fixc=0/1, with and without W1; output equality PASSes
  everywhere. NOT cross-case contamination (balanced is the first case;
  run A precedes any W2-on run). Genuine W2-on backward delta: forward
  bitwise, input grads not. Suspects: chunked dispatch/combine reverse-A2A
  grad views; grouped-GEMM dgrad under zero-padded num_gemms (the R1
  residual the gate exists to settle); probs-path excluded (router
  monkeypatched in the gate ⇒ probs constant ⇒ input grad flows through the
  token path only).
- **Defect 2 — cross-case state leak:** imbalance run A (W1-off/W2-off
  MONOLITHIC) hard-crashed all ranks at `token_dispatcher.py:1930`
  all_to_all — "Split sizes doesn't match total dim 0 size". Leading
  hypothesis: v3 leaves per-instance state after W2-on cases that the
  monolithic branch reads (`set_w2(False)` only nulls `_w2_config`). If
  confirmed, PRODUCTION-relevant: the runtime fallback/disarm path must
  restore a clean monolithic pass.
- **Harness thread — CLOSED (fourier, corrected):** box gate .py md5
  e86589de (333 lines) vs Mac f475e2e7 (335); diff = exactly the
  `BT_T2_SKIP_FIXC` conditional (box: older 1-line unconditional
  fixc_cases). With the SKIP export removed the box revision ran fixc
  variants unconditionally = intended inclusion; result unconfounded. The
  box copy DOES pop the fixc env (box line 320) — fourier's earlier
  "never pops" was a sed-window artifact; the cross-case env-leak channel
  does NOT exist. Sweep-scope gap recorded for kepler: estate sweep covered
  patch files only, never test/tooling scripts (kepler extending).
- **Discipline rule (kepler, binding):** defect-1 treated as
  UNCONFIRMED-REAL until Mac gloo repro reproduces it or a verified-fresh
  box harness re-run does. Prove-don't-assume before sinking into v3's
  backward.
- **Evidence artifacts:** box `lps1062_bench/t2_w2_gate_fixc_included_FAIL_0626.log`
  (md5 7ba61009) + Mac `~/perf_profiles/lps-1062/round3/` same;
  `token_dispatcher.py.pre-v3.orig` (pre-v3 tree state) and the 04:25 v1
  FAIL log `t2_w2_gate_FAIL_318g61w.log` for signature diffing.
- **Box state:** v3 REVERSED off the tree pre-canary (kepler's order;
  Phase-1 zero-marker criteria as the reversal check). Box repro of this
  FAIL needs v3 re-applied — sequenced after the W3 canary readout.

## 2026-08-09 — rulings carried (from kepler ratifications)

- T2 gate runs WITH fixc variants (no `BT_T2_SKIP_FIXC`); fixc branches
  self-arm `BT_MOE_DISPATCH_REPLAY_CACHE` in-process (lazy per-call read,
  token_dispatcher.py:123); the one-time gate-state WARNING latches DISABLED
  on the first (non-fixc) call — **stats asserts are the sole admissible
  evidence** on fixc variants (the armed-line discriminator retracted).
- Loss verdicts (boltzmann): in-process gates hard-bitwise; cross-boot
  canaries house band ≤2e-3 pass / >5e-3 stop; matched `--warmup-datums`
  mandatory — mismatch = INVALID comparison (re-run), not FAIL; warmup0 +
  warmup-datums recorded in every arm JSON.
- W3 canary binding: sweeps==0 halt rule; 16k×d32 gated OFF until measured;
  FIX C paired; mechanism-first acceptance per two-tier verdicts.
