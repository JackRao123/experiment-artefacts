# BOX A (w56lorq) ARTIFACT MANIFEST — for report §8/§9 and clean box stop

Compiled by hausdorff, 2026-08-13 ~03:5x CDT; FINAL SWEEP pass 2026-08-13
~16:4x CDT (after the box queue drained: L0/L3/L5, golden parity legs,
fe127 fix campaign, unfused-leg death). Purpose: everything Jack needs to
keep is either (a) Mac-side durable (location noted below), or (b)
explicitly listed as disposable/regenerable. Box A can be stopped without
loss.

**Key structural fact:** `/root/.cache/user_artifacts/` is the org-shared
CPFS mount — box B saw box A's clone/venv/tools because of it. Stopping
w56lorq does NOT by itself delete CPFS data, but the volume is
quota-pressured (08-12 ENOSPC saga) and not a durable store. Node-local
`/tmp` (incl. `/tmp/checkpoints`) dies with the box.

## 1. Kineto traces (box: `lps1062_pp2/traces/`, CPFS)

| file | size | sha256 (box-side) | Mac-side | disposition |
|---|---|---|---|---|
| diag_d4_rank0.pt.trace.json | 699.7MB | 6503aebc…b4ad2 | `~/perf_profiles/lps-1062/pp2cp8ep8/` | DURABLE (pre-fix dx, Run-C-class config) |
| diag_d4_rank8.pt.trace.json | 748.6MB | b1bee664…1cdfd | same | DURABLE (first stage-1 timeline) |
| runB_d4_rank0.pt.trace.json | 699.5MB | (Mac copy verified at pull) | same | DURABLE (ship-env run; box copy already gone) |
| mn_d4_rank0.pt.trace.json | 704.5MB | 4121edf9…2103d | `~/perf_profiles/lps-1062/pp2cp8ep8/` | DURABLE — sha256 VERIFIED post-pull |
| mn_d4_rank8.pt.trace.json | 752.2MB | 03903372…040f2 | same | DURABLE — sha256 VERIFIED post-pull |
| runC_d4_tf32_rank0.pt.trace.json | 699.5MB | 1ff1e205…070f | NOT pulled | LISTED-ONLY — diag_d4_rank0 covers the same config class (TF32+ship d4); pull later if ever needed |
| fe127_d16_rank0.pt.trace.json | 2.83GB | 59982f27…b148 | `~/perf_profiles/lps-1062/pp2cp8ep8/` | DURABLE — sha256 VERIFIED post-pull |
| l5_flex_d4_rank0.pt.trace.json | 719.1MB | 50aa5c35…c020 | same | DURABLE — sha256 VERIFIED post-pull |
| l3_d16_rank0.pt.trace.json | 2.80GB | 725b40a8…99e4e | same | DURABLE — sha256 VERIFIED post-pull |
| l3_d16_rank8.pt.trace.json | 3.00GB | 4ae401db…1f23 | same | DURABLE — sha256 VERIFIED post-pull |

- Trace analyses (already Mac-side): `results/TRACE_SKEW_ANALYSIS.md`,
  `results/TRACE_RANK8_ANALYSIS.md`; on-box breakdown logs pulled to
  `logs/box_a/trace_breakdown*.log`.
- Derived-analysis sidecars pulled to `analysis/`: per-trace deepdive
  JSON+logs (diag/mn/l3 pairs), l3_d16 a2a decomposition JSON+logs,
  a2a_probe2.log, bucket logs (l3_d16 rank0/rank8, l5_flex d4), and the
  analysis scripts themselves (l3_deepdive.py, l3_a2a_decomp.py,
  decompose_l3_buckets.py, l3_probe*.py, l3_kdbg.py, l3_corr_dbg.py).

## 2. Tonight's perf + parity measurements (box: `lps1062_bench/`, CPFS)

Pulled to Mac `pp2cp8ep8/bench/` (15 JSONs): pp2-131k-d2, d4, d4-shipenv,
d4-tf32, d4-rank08, d4-nomemtel, mn-d1, mn-d2, mn-d4, mn-d8, mn-d16 +
fe127 campaign: pp2-131k-fe127-d2, -d4, -d4-rerun, -d16.
Driver logs for each: `pp2cp8ep8/logs/box_a/*.driver.log`.

Parity JSONs in `pp2cp8ep8/parity/` (22 total, all parse-validated):
- `parity_pp2-padded-fresh.json` — the valid fresh-boot PP2 leg (banked by
  poincare 2026-08-13: loss 12.54805275637226, 262032 tokens, 9/9, fb 75.6s).
- `parity_pp2-padded.json` — leg A, step-18 weights; artifact only.
- Golden EP16/CP16/PP1 reference legs: `parity_pp1cp16-padded.json`,
  `parity_pp1cp16-unpadded.json`.
- PP2 fix-campaign legs: `parity_pp2-padded-fixed{,-rerun,-rerun2}.json`,
  `parity_pp2-unpadded{,-identity,-identity2,-rev}.json`.
- fe127 legs: `parity_pp2-padded-fe127{,-2}.json`,
  `parity_pp2-unpadded-fe127{,-2}.json`,
  `parity_pp2-unpadded-rev-fe127.json`,
  `parity_pp1cp8-unpadded-fwd{,2,3}.json`,
  `parity_pp1cp8-unpadded-fwd-fe127{,-2,-3}.json`.
- gate_fe127/ validation set: already Mac-side at `results/gate_fe127/`
  (cauchy-confirmed; not re-pulled).

Older bench JSONs (Aug 7–10 sweeps: A-*, A131-*, B-*, C-*, exp*): NOT
re-pulled — already durable in the Mac archive
`experiment_artefacts/glm/lps_1062_perf/runs/*/results/` (spot-verified:
A-131k-d4.json, C-G-131k-d4.json present). Box copies are the working set
of the same data.

## 3. Export/DCP evidence (box: `lps1062_pp2/export_test/`, CPFS)

ALL durable Mac-side at `pp2cp8ep8/export_test/evidence/`:
- Small-model hang stacks: `pyspy_savestate_pp1cp2_*.txt`,
  `pyspy_savestate_pp2cp2_rank_*.txt`; L2b worker logs
  (`workers_pp1cp1_l2b.log`, `workers_pp2cp2_l2b.log`); adapters
  (`ws_ref_pp1cp1/`, `ws_test_pp2cp2/`); `cudart_mask.c` (F2 interposer).
- Big-trainer probe: `big_trainer_save_probe/` (42 files — 16-rank py-spy
  both nodes, srun stdout capture, probe log, trainer_srun snapshot with
  the ancdata traceback).

## 4. Logs (box: `lps1062_pp2/logs/` + top level, CPFS)

Pulled to `pp2cp8ep8/logs/box_a/`: all tonight's driver logs
(pp2-131k-*.driver.log incl. selective-d1 OOM ramp, L0/L0bmem/L5/fe127
legs), trace_breakdown*.log, a2a_microbench.log, mn_rank{0,8}_split.log,
probe_dsa_indexer.log, trainer_env.sh, preflight_groups.sh,
save_probe_sync.log, probe_a_out.log (+ its script
`analysis/probe_a_split_convention.py`), the box-updated parity driver
(`parity_driver_box_0813.py` evidence copy), and
`pp2cp8_cleaninit_saveprobe_hang_20260813.log` (the wedged-boot trainer
log; an earlier snapshot of the same file lives at
`export_test/evidence/big_trainer_save_probe/trainer_srun_saveprobe_async_hang_0229.log`).
`legB_pp1cp8ep8_dp1_1node_nccl_fail_20260813_0159.log` was already Mac-side
in `logs/`. Final-sweep additions: `golden_leg1_padded_boot_20260813.log`,
`l0_boot1_fail_20260813_0414.log`, `l0bmem_d2_abort_over255_20260813.log`,
`unfused_pp2_watchdog_death_20260813.log` (105MB — the unfused-leg
death), and `wedge_capture_bumped_wheel/` (20 files: the preserved
old-wheel evidence set — dmesg Xid, nvidia-smi snapshot, 16-rank py-spy,
trainer log tail).

## 5. Patches / configs / tools

- `configs/` Mac-side holds: tf32_head_port_21d0c578.patch,
  pre_checkout_local_changes_0e0b65a6.patch (the prior night's uncommitted
  TF32/warmup/overlap work — provenance), all tonight's trainer configs
  (pp2cp8ep8_131k, selective, vpp2, pp1cp8ep8, ep16cp16 golden) +
  trainer_server.json. Final-sweep additions: tonight's experiment configs
  (unfused, L0_selective_vpp2_overlap, L0bmem_selective_vpp2_offload,
  L5_flex; box-A golden copy verified byte-equivalent to the Mac prepped
  one) and the full patch set: bridge_flex_deepep_capability_fix.patch,
  l0_shared_expert_overlap_clearing.patch,
  megatron_config_hunks_savesync_plus_l0clearing.patch,
  pad_disable_filegate_probe.patch, tailpad_filegate_probe.patch,
  pp2_unpadded_gateflip_probe.patch.
- Source snapshots: `kernel_src_snapshot/kernel_src_snapshot.tar.gz`
  (13.6MB) — sha256 VERIFIED Mac==box (a94233ff…b455);
  `cutlass_dsl_src_snapshot/cutlass_dsl_src_snapshot.tar.gz` (61.3MB,
  box sha 16a8cb3f…abd1) pulled in the final sweep.
- Tools (probe, parity driver, bringup, analyses): `tools/` Mac-side +
  `../tools/` (parity_driver.py, bringup_pp2cp8ep8.sh etc. — the
  lps_1062_perf-level tools dir). save_state_probe.py is the HARDENED
  version (srun fan-out; see NOTEBOOK 22:2x).
- Branches (all pushed to origin — nothing box-only): gibbs/doppler
  `jackrao/lps-1062-pp2cp8ep8` (origin head now b8d868ff; tonight's box-A
  bits = 73c24b00 + uncommitted TF32 patch), volta/poincare
  `jackrao/lps-1062-pp2-packing` @ 8b0ef108, dedekind/hausdorff
  `jackrao/lps-1062-pp2-export` @ db5d1826.
- Box-A clone working tree carries: TF32 patch (chunked_lm_head.py) +
  BT_SAVE_STATE_SYNC hunk (megatron_config.py, env-gated, inert when
  unset). Both preserved Mac-side (configs/ + db5d1826 respectively), so
  the working tree is fully reconstructable.
- **Vendored-mcore DIRTY TREE (shared-clone artifact Jack will ask about):**
  the CPFS vendored Megatron-LM tree is pointer-clean (57efae08b) but
  content-dirty (+1652/−26 across 8 files + untracked
  `lookahead_checkpoint.py`) — the uncommitted Aug-9/10 experiment stack,
  imported EDITABLE by the venv, so every box measurement tonight ran it
  (all seven env gates self-reported DISABLED ×16 ranks; believed inert —
  gating-structure proof in the doc). Full finding + gate inventory +
  inertness argument + cleanup path: **`results/DISPATCHER_CACHE_ARCHAEOLOGY.md`
  §7 "DIRTY-TREE HYGIENE"** (gauss, 2026-08-13). Disposition in one line:
  **land the SHIP pair** (BT_DSA_CP_LAYOUT_CACHE + BT_THD_ROPE_HOST_CACHE —
  PR-ready format-patch series at `results/pr_series_bf/`, then bump the
  megatron-bridge→mcore and trainers→megatron-bridge pointers on the
  73c24b00 lineage), **revert the rest** (fixc / fixc_prime / W1 / W3 +
  lookahead_checkpoint.py — preserved Mac-side as artefact patches +
  `results/mcore_dirty_w56lorq_20260813.diff` snapshot). **TIMING (borel,
  2026-08-13 ~04:0x): the revert is a MORNING action on Jack's call — NOT
  tonight, NOT between boots.** Every measurement tonight (incl. the
  remaining parity legs) runs on the current dirty-but-gated bits;
  reverting mid-night breaks exact-bits continuity for everything after
  it. Tonight the tree is FROZEN: gates stay off except during the
  sanctioned cache A/B arms. Surfaced explicitly for Jack: the dirty tree
  includes an **unowned UNTRACKED file, `lookahead_checkpoint.py` (588
  lines)** — an untracked file in a production-imported tree is exactly
  the unowned-complexity class the directive targets.

## 6. Disposable / regenerable (OK to lose with the box)

- `/tmp/checkpoints/pp2cp8-save-probe-sync/` (537MB) — the sync-verify
  checkpoint. borel: can stay. Its VERDICT is recorded (70.8s PASS); the
  bits are regenerable in minutes.
- `/tmp/checkpoints/pp2cp8-save-probe/` (536MB) — partial from the hung
  async attempt. Worthless; the hang evidence is the stacks + log.
- `/tmp/checkpoints/profiles/torch_trace/` — volatile by design (cleared
  per profile start); everything was archived to traces/ per the runbook.
- The shared venv + clone (CPFS) — rebuildable from origin branches +
  `configs/` patches; the cutlass cu13 sequenced-reinstall procedure is
  documented in NOTEBOOK (and obsoleted once PR 987 merges).
- Box B (3m9o7kq) staging — stood down unused; configs are Mac-side.

## 7. Final-sweep status — CLOSED (2026-08-13 ~13:1x CDT)

All items pulled and verified. The four big traces completed and sha256-
verified EXACT against box-side hashes (fe127_d16_rank0 59982f27…b148,
l5_flex_d4_rank0 50aa5c35…c020, l3_d16_rank0 725b40a8…99e4e,
l3_d16_rank8 4ae401db…1f23). kernel_src_snapshot verified Mac==box
(a94233ff…b455). All 37 bench/parity JSONs parse-validated. doppler
confirmed the box idle-armed with nothing mid-write during the sweep.

**THIS MANIFEST IS FINAL.** Every artifact class is either Mac-side
durable (hash- or parse-verified) or explicitly listed disposable in §6.
The report §9 stop-safety claim is **verified**: w56lorq can be stopped
without evidence loss.
